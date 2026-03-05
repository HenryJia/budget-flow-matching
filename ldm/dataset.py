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
        self.backoff_time = 2.0

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
            try:
                response = requests.get(url, headers=self.request_header)

                if response.status_code == 200:
                    self.backoff_time = 2.0 # Reset backoff time after a successful request
                elif response.status_code == 429: # Too Many Requests
                    print(f"Received 429 Too Many Requests for image {idx}, backing off for {self.backoff_time} seconds")
                    time.sleep(self.backoff_time)
                    self.backoff_time = min(self.backoff_time * 2, 32) # Exponential backoff
                    return self.__getitem__(idx) # Try again after backing off
                else: # Any other error
                    raise ValueError(f"Failed to download image {idx} from {url}, response code: {response.status_code}")

                image = Image.open(BytesIO(response.content))

                image.save(os.path.join(self.img_dir, f"{idx}.png")) # Cache the image so we don't have to download it again in the future
            except Exception as e:
                print(f"Failed to download image {idx} from {url}, error: {e}. Skipping and loading the next one instead")
                return self.__getitem__((idx + 1) % len(self.dataset_hf)) # Just skip this image and try the next one
        
        image = image.convert("RGB")
        h, w = image.size
        size = torch.tensor([h, w]) / 256.0 # just to keep numbers small
        if self.transform:
            image = self.transform(image)
        return image, item["description"], size