import os
import time
import requests
import warnings
import lzma
import re
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO

import torch

import torchvision as tv
from torch.utils.data import Dataset, IterableDataset

import datasets
import webdataset as wds

from rich.progress import Progress

class HFDataset(Dataset):
    def __init__(self,
        dataset_name, img_key, text_key,
        split="train", transform=None, img_dir=None, min_backoff_time=0.25):

        self.dataset_hf = datasets.load_dataset(dataset_name, split=split)
        self.transform = transform
        self.img_dir = img_dir
        self.min_backoff_time = min_backoff_time
        self.img_key = img_key
        self.text_key = text_key

        if self.img_dir is not None and not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.backoff_time = self.min_backoff_time

    def __len__(self):
        return len(self.dataset_hf)

    def __getitem__(self, idx):
        item = self.dataset_hf[idx]
        img = item[self.img_key]

        # Check if HF already provides the image
        if isinstance(img, Image.Image):
            image = img.convert("RGB")

        elif isinstance(img, str): # Assume it's a URL
            url = img
            if os.path.exists(os.path.join(self.img_dir, f"{idx}.jpg")):
                try:
                    image = Image.open(os.path.join(self.img_dir, f"{idx}.jpg"))
                    image = image.convert("RGB")
                except Exception as e:
                    warnings.warn(f"Failed to load cached image {idx}, redownloading. Error: {e}", ResourceWarning)
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
                        warnings.warn(f"Received 429 Too Many Requests for image {idx}, backing off for {self.backoff_time} seconds")
                        self.backoff_time = min(self.backoff_time * 2, 32) # Exponential backoff
                        return self.__getitem__(idx) # Try again after backing off
                    else: # Any other error
                        raise ValueError(f"Failed to download image {idx} from {url}, response code: {response.status_code}")

                    image = Image.open(BytesIO(response.content))
                    image = image.convert("RGB")

                    image.save(os.path.join(self.img_dir, f"{idx}.jpg")) # Cache the image so we don't have to download it again in the future
                except Exception as e:
                    warnings.warn(f"Failed to download image {idx} from {url}, error: {e}. Skipping and loading the next one instead", ResourceWarning)
                    return self.__getitem__((idx + 1) % len(self.dataset_hf)) # Just skip this image and try the next one
        else:
            warnings.warn(f"Unexpected image format for item {idx}: {type(img)}. Skipping and loading the next one instead", ResourceWarning)
            return self.__getitem__((idx + 1) % len(self.dataset_hf)) # Just skip this image and try the next one instead
        
        h, w = image.size
        size = torch.tensor([h, w]) / 256.0 # just to keep numbers small
        if self.transform:
            image = self.transform(image)

        if image is None or item[self.text_key] is None:
            warnings.warn(f"Failed to load image {idx} from {url}. Got None instead. Skipping and loading the next one instead", ResourceWarning)
            return self.__getitem__((idx + 1) % len(self.dataset_hf)) # Just skip this image and try the next one instead

        return idx, image, item[self.text_key], size


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir

        filter_pattern = re.compile(r"(\d+)_precalc\.pt.xz")

        self.file_list = os.listdir(self.embedding_dir)
        self.file_list = [f for f in self.file_list if filter_pattern.match(f)]

    def __len__(self):
        return 4096 #len(self.file_list)

    def __getitem__(self, idx):
        try:
            with lzma.open(os.path.join(self.embedding_dir, self.file_list[idx]), "rb") as f:
                precalc = torch.load(f)
            return precalc
        except Exception as e:
            warnings.warn(f"Failed to load embedding {idx} from {self.file_list[idx]}, error: {e}. Skipping and loading the next one instead")
            return self.__getitem__((idx + 1) % len(self.file_list))


class PD12MFullDataset(IterableDataset):
    def __init__(self, root_dir, transform=None, shuffle_buffer=None):

        self.wds = wds.WebDataset(os.path.join(root_dir, "{00000..02480}.tar"))
        if shuffle_buffer is not None:
            self.wds = self.wds.shuffle(shuffle_buffer)
        self.wds = self.wds.decode("pil")

        self.transform = transform

    def __len__(self):
        # Annoyingly webdataset doesn't give us an easy way to get the length
        return 12400094 # Maximum possible length based on the number of rows of the parquet metadata

    def __iter__(self):
        wds_iter = iter(self.wds)
        for idx in range(len(self)):
            try:
                item = next(wds_iter)
            except StopIteration:
                break # End of the dataset
            image = item["jpg"]
            image = item["jpg"]
            text = item["txt"]

            h, w = image.size
            size = torch.tensor([h, w]) / 256.0 # just to keep numbers small

            if self.transform:
                image = self.transform(image)

            yield idx, image, text, size


class CombinedDatasetWrapper(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = []
        cumulative_length = 0
        for dataset in self.datasets:
            cumulative_length += len(dataset)
            self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.cumulative_lengths[dataset_idx]:
            dataset_idx += 1
        if dataset_idx > 0:
            idx -= self.cumulative_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][idx]