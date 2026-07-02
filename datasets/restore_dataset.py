import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

class SyntheticSRDataset(Dataset):

    def __init__(self, num_samples=200, hr_size=224, patch_size=48, scale=1):
        self.num_samples = num_samples
        self.hr_size = hr_size
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # HR image synthetic
        hr_full = torch.rand(3, self.hr_size, self.hr_size)

        # random crop
        x = random.randint(0, self.hr_size - self.patch_size)
        y = random.randint(0, self.hr_size - self.patch_size)

        hr = hr_full[:, y:y+self.patch_size, x:x+self.patch_size]

        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1/self.scale if self.scale > 1 else 1,
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        return lr, hr
