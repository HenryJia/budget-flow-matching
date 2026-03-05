import os
import time
import requests
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO

import torch

import torchvision as tv
from torch.utils.data import Dataset

import datasets

class PublicDomainDataset(Dataset):
    def __init__(self, split="train", transform=None, img_dir="../publicdomain_imgs"):
        self.dataset_hf = datasets.load_dataset("nyuuzyou/publicdomainpictures", split=split)
        self.transform = transform
        self.img_dir = img_dir

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.request_header = {
            'User-Agent': 'Henry\'s Diffusion and Flow Project'
            }

    def __len__(self):
        return len(self.dataset_hf)

    def __getitem__(self, idx):
        item = self.dataset_hf[idx]
        url = item["image_url"]

        if os.path.exists(os.path.join(self.img_dir, f"{idx}.png")):
            try:
                image = Image.open(os.path.join(self.img_dir, f"{idx}.png"))
            except Exception as e:
                print(f"Failed to load cached image {idx}, redownloading. Error: {e}")
                os.remove(os.path.join(self.img_dir, f"{idx}.png"))
                return self.__getitem__(idx) # Try again, this time it will download the image instead of loading from cache
        else:
            time.sleep(2.0) # Be nice to the server and don't send requests too quickly
            response = requests.get(url, headers=self.request_header)

            if response.status_code != 200:
                raise ValueError(f"Failed to download image {idx} from {url}, response code: {response.status_code}")

            image = Image.open(BytesIO(response.content))

            image.save(os.path.join(self.img_dir, f"{idx}.png")) # Cache the image so we don't have to download it again in the future
        
        image = image.convert("RGB")
        h, w = image.size
        size = torch.tensor([h, w]) / 256.0 # just to keep numbers small
        if self.transform:
            image = self.transform(image)
        return image, item["description"], size