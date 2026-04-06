import os
import argparse
import pandas as pd
import lzma
import copy
import warnings
#warnings.filterwarnings("ignore", category=ResourceWarning) # Suppress resource warnings from the dataset
from threading import Thread, Event
from queue import Queue

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.utils.data as data
import torchvision as tv

from diffusers import AutoencoderDC
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel

from dataset import HFDataset, PD12MFullDataset

from model import PromptEncoderWrapper, ViTWrapper

from rich.progress import Progress

def lzma_thread(batch_queue, output_dir, gpu_done):
    while not gpu_done.is_set() or not batch_queue.empty():
        batch = batch_queue.get()
        idxs, dcae_embedding, repa_embedding, prompt_embedding, prompt_mask, size = batch
        # Unbatch and save the embeddings to disk as .pt files
        for i, idx in enumerate(idxs):
            with lzma.open(os.path.join(output_dir, f"{idx}_precalc.pt.xz"), "wb", preset=9) as f:
                torch.save({
                    'dcae_embedding': dcae_embedding[i],
                    'repa_embedding': repa_embedding[i],
                    'prompt_embedding': prompt_embedding[i],
                    'prompt_mask': prompt_mask[i],
                    'size': size[i],
                }, f)

def gpu_thread(batch_queue, device, dcae, repa_model, prompt_encoder, output_dir, loader_done):
    lzma_queue = Queue()
    gpu_done = Event()
    gpu_done.clear()
    save_thread = Thread(target=lzma_thread, args=(lzma_queue, output_dir, gpu_done))
    save_thread.start()

    while not loader_done.is_set() or not batch_queue.empty():
        batch = batch_queue.get()
        idxs, img, caption, size = batch
        with torch.no_grad():
            img = img.half()
            img = img.to(device)

            # Note, as far as I'm aware from the Huggingface diffusers source code, they DO NOT apply the scaling factor for us
            # We have to do it ourselves to ensure the magnitudes are correct for the velocity prediction task
            dcae_embedding = dcae.encode(img).latent * dcae.config.scaling_factor
            repa_embedding = repa_model((img + 1.0) / 2.0 * 255.0) # Rescale for DinoV2, which I think expects things in [0, 255]
            repa_embedding = torch.mean(repa_embedding, dim=1) # Take the mean over the spatial dimension to save memory
            prompt_embedding, prompt_mask = prompt_encoder(caption)

            lzma_queue.put((idxs, dcae_embedding.cpu(), repa_embedding.cpu(), prompt_embedding.cpu(), prompt_mask.cpu(), size))
    gpu_done.set()
    save_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for precomputing embeddings.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--devices", type=str, nargs='+', help="Devices to use for computation.")
    parser.add_argument("--output_dir", type=str, default="./precalc_embeddings", help="Directory to save the precomputed embeddings.")

    args = parser.parse_args()


    input_dim = (256, 256)
    input_channels = 3
    latent_dim = (8, 8)
    latent_channels = 32
    if args.dataset == "nyuuzyou":
        # nyuuzyou/publicdomainpictures is relatively high quality but it is smaller (600k)
        dataset = HFDataset(
            dataset_name="nyuuzyou/publicdomainpictures", img_key="image_url", text_key="description", 
            split="train", img_dir='~/publicdomain_imgs', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )
    elif args.dataset == "coco":
        # Use a version of COCO that's been helpfully preprocessed by someone else on Huggingface
        dataset = HFDataset(
            dataset_name="jxie/coco_captions", img_key="image", text_key="caption",
            split="train", img_dir='~/coco', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )
    elif args.dataset == "sbu":
        # SBU Captions dataset is supposed to be 1M images from Flickr with real captions. We only managed to get 850k of them because the rest were missing
        # We'll use a version on Huggingface that's been helpfully preprocessed
        dataset = HFDataset(
            dataset_name="eaglewatch/sbucaptions", img_key="url", text_key="caption",
            split="train", img_dir='~/SBU_Captions', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )
    elif args.dataset == "pd12m-full":
        dataset = PD12MFullDataset(
            root_dir='/mnt/pd12m-full/webdataset', transform=tv.transforms.Compose([
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

    workers = args.num_workers if args.dataset != "pd12m-full" else 1 # Iterable dataset will duplicate our dataset with more than 1 worker
    dataloader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers, pin_memory=True, prefetch_factor=4)

    devices = [torch.device(d) for d in args.devices]

    dcae = [copy.deepcopy(dcae).to(d) for d in devices]
    repa_model = [copy.deepcopy(repa_model).to(d) for d in devices]
    prompt_encoder = [copy.deepcopy(prompt_encoder).to(d) for d in devices]

    print(f"Precalculating embeddings for {len(dataset)} items in the {args.dataset} dataset using devices {args.devices} and saving to {args.output_dir}...")
    print(f"Total of {len(dataloader)} batches with batch size {dataloader.batch_size}.")

    loader_done = Event()
    loader_done.clear()
    gpu_queue = [Queue(maxsize=32) for _ in devices]

    threads = []
    for i, d in enumerate(devices):
        threads.append(Thread(target=gpu_thread, args=(gpu_queue[i], d, dcae[i], repa_model[i], prompt_encoder[i], args.output_dir, loader_done)))
        threads[-1].start()

    with Progress() as progress:
        task = progress.add_task("[cyan]Precomputing embeddings...", total=len(dataloader))
        for batch_idx, batch in enumerate(dataloader):
            device_idx = batch_idx % len(devices)
            gpu_queue[device_idx].put(batch)

            progress.update(task, advance=1)

    print("Almost done! Waiting for remaining threads to finish...")
    loader_done.set()
    for t in threads:
        t.join()
    print("All done!")