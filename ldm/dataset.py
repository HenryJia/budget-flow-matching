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

from rich.progress import Progress

class PublicDomainDataset(Dataset):
    def __init__(self, split="train", transform=None, img_dir="../publicdomain_imgs", min_backoff_time=0.25):
        self.dataset_hf = datasets.load_dataset("nyuuzyou/publicdomainpictures", split=split)
        self.transform = transform
        self.img_dir = img_dir
        self.min_backoff_time = min_backoff_time

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.backoff_time = self.min_backoff_time

    def __len__(self):
        return len(self.dataset_hf)

    def __getitem__(self, idx):
        item = self.dataset_hf[idx]
        url = item["image_url"]

        if os.path.exists(os.path.join(self.img_dir, f"{idx}.jpg")):
            try:
                image = Image.open(os.path.join(self.img_dir, f"{idx}.jpg"))
                image = image.convert("RGB")
            except Exception as e:
                print(f"Failed to load cached image {idx}, redownloading. Error: {e}")
                os.remove(os.path.join(self.img_dir, f"{idx}.jpg"))
                return self.__getitem__(idx) # Try again, this time it will download the image instead of loading from cache
        else:
            try:
                request_header = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0',
                    'Cookie': 'jazyk=EN; cf_clearance=txORAymedBE0l0UwhGtDJZD6YhE8eE5UcTM1ATrlYLM-1772803347-1.2.1.1-1cSa3faybR3t.BpBaKqorm_JM5lZT0KMHK5s3V9sQnCUa3fCfYlDzNRmdTF0XwC.djhdRfljrWrndxviquDOzG2u2oUBk8kws2AZRoKKV1Qb.Zmyi247qmqKhSLbWkMIFOvA3DHO8.mtuG5Yqif.o.90qpqm3qb6FUK6mHj1Mv.16tiyvI3roBjoCu3aSDLYPoBDVpVz9rgMpTP_EvCgnBbwoFoSMfQpZnjrh3uyuWF64N4Nw1iLXXSxfid9o8NE; PHPSESSID=fa902799b6523a9a1205a06b59ea8934'
                    }
                
                response = requests.get(url, headers=request_header)
                if response.status_code == 200:
                    self.backoff_time = self.min_backoff_time # Reset backoff time after a successful request
                elif response.status_code == 429: # Too Many Requests
                    time.sleep(self.backoff_time)
                    print(f"Received 429 Too Many Requests for image {idx}, backing off for {self.backoff_time} seconds")
                    self.backoff_time = min(self.backoff_time * 2, 32) # Exponential backoff
                    return self.__getitem__(idx) # Try again after backing off
                else: # Any other error
                    raise ValueError(f"Failed to download image {idx} from {url}, response code: {response.status_code}")

                image = Image.open(BytesIO(response.content))
                image = image.convert("RGB")

                image.save(os.path.join(self.img_dir, f"{idx}.jpg")) # Cache the image so we don't have to download it again in the future
            except Exception as e:
                print(f"Failed to download image {idx} from {url}, error: {e}. Skipping and loading the next one instead")
                return self.__getitem__((idx + 1) % len(self.dataset_hf)) # Just skip this image and try the next one
        
        h, w = image.size
        size = torch.tensor([h, w]) / 256.0 # just to keep numbers small
        if self.transform:
            image = self.transform(image)
        return image, item["description"], size