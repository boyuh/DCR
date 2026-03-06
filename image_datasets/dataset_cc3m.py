import os
from typing import Optional, Sequence, Dict, Union, Tuple
from dataclasses import dataclass, field
from functools import partial
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from datasets import load_dataset
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_IMAGE_FILE_SUFFIX = ['jpg', '0.jpg', '0.png', 'png', 'jpeg', '0.jpeg', 'webp']


def find_image(sample):
    for suffix in DEFAULT_IMAGE_FILE_SUFFIX:
        if suffix in sample.keys():
            sample['0.jpg'] = sample[suffix]
            break
    return sample


def _convert_to_rgb(image):
    try:
        image = image.convert('RGB')
    except Exception as e:
        print(e)
    return image


def to_tensor(image):
    try:
        image = ToTensor()(image)
    except Exception as e:
        image = image.float()
        print(e)
    return image


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


def image_transform(
    image_size: Union[int, Tuple[int, int]],
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            # ToTensor(),
            to_tensor,
        ])
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            # ToTensor(),
            to_tensor,
        ])


def collate_anyres(images, sizes, patch_size, max_size=2048):
    """
    Args:
    * images: list of images
    * sizes: list of image sizes in (ph, pw), i.e., number of patches in h and w
    
    Return: args accepted by VQModel
    * pixel_values: packed images
    * cu_seqlens_img
    * max_seqlen_img
    * grid_hw
    * image_sizes
    """
    b, c = len(images), images[0].shape[0]
    max_patch_num = max_size // patch_size

    image_sizes = torch.tensor([(image.shape[1], image.shape[2]) for image in images])
    H, W = image_sizes.max(dim=0).values
    padded_images = images[0].new_zeros(size=(b, c, H.item(), W.item()))

    h, w = torch.tensor(sizes).max(dim=0).values
    padding_masks = torch.zeros(size=(b, h.item(), w.item()), dtype=torch.bool)

    for i, (image, mask_size) in enumerate(zip(images, sizes)):
        padded_images[i, :, : image.shape[1], : image.shape[2]].copy_(image)
        padding_masks[i, : mask_size[0], : mask_size[1]] = 1

    padded_images = padded_images.reshape(b, c, h, patch_size, w, patch_size)
    padded_images = torch.einsum("nchpwq->nhwpqc", padded_images)
    padded_images = padded_images.reshape(b, h, w, -1)
    packed_images = padded_images[padding_masks]

    seq_lens = padding_masks.flatten(1, 2).sum(dim=-1)
    cu_seqlens_img = torch.nn.functional.pad(
        torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seqlen_img = seq_lens.max()

    grid_h = torch.arange(0, h)[None, :, None].repeat(b, 1, w)
    grid_w = torch.arange(0, w)[None, None, :].repeat(b, h, 1)
    grid_hw = grid_h[padding_masks] * max_patch_num + grid_w[padding_masks]
    
    return packed_images, cu_seqlens_img, max_seqlen_img, grid_hw, torch.tensor(sizes)


def get_cc3m_wds_dataset_and_collator(img_size, img_dir, seed, patch_size):
    train_processor = image_transform(img_size, is_train=True)
    val_processor = image_transform(img_size, is_train=False)

    data = load_dataset("webdataset", data_dir=img_dir, split="train", streaming=True)
    data = data.shuffle(buffer_size=2_000, seed=seed)

    def decode(sample, img_processor):
        sample = find_image(sample)
        sample['image'] = img_processor(sample['jpg'])
        sample['text'] = sample['txt']
        return sample
    data = data.map(
        partial(decode, img_processor=train_processor),
        remove_columns=['__key__', '__url__']
    )
    data = data.filter(lambda sample: 'image' in sample and 'text' in sample) # filter return samples that match the given condition
    data_collator = CC3M_WebdatasetCollator(patch_size)

    return data, data_collator


@dataclass
class CC3M_WebdatasetCollator:
    def __init__(self, patch_size: int = 1):
        self.patch_size = patch_size
        self.count = 0

    def __call__(
        self, 
        samples: Sequence[Dict],
        ) -> Dict[str, torch.Tensor]:

        self.count += 1
        images = [sample["image"] for sample in samples]
        texts = [sample["text"] for sample in samples]

        if "size" in samples[0]:
            sizes = [sample['size'] for sample in samples]

        batch = {}

        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['image'] = torch.stack(images)
        else:
            batch['image'] = images
        batch['text'] = texts
        return batch



def loader(train_batch_size, num_workers, **args):
    cc3m_dataset, cc3m_collator =  get_cc3m_wds_dataset_and_collator(**args)
    return DataLoader(cc3m_dataset, batch_size=train_batch_size, num_workers=num_workers, collate_fn=cc3m_collator)
