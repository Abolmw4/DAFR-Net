import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DAFRDataSet(Dataset):
    def __init__(self, gt_dir, aug_dir, transform=None):
        self.gt_dir = gt_dir
        self.aug_dir = aug_dir
        self.transform = transform if transform else transforms.ToTensor()
        
        self.gt_map = {}
        for filename in os.listdir(gt_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name_without_ext, _ = os.path.splitext(filename)
                self.gt_map[name_without_ext] = os.path.join(gt_dir, filename)
        
        
        self.valid_pairs = []
        for filename in os.listdir(aug_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                aug_path = os.path.join(aug_dir, filename)
                
                gt_key = None
                if "Last_" in filename:
                    gt_key = filename.split("Last_")[0] + "Last_"
                elif "First_" in filename:
                    gt_key = filename.split("First_")[0] + "First_"
                if gt_key and gt_key in self.gt_map:
                    self.valid_pairs.append({
                        'aug_path': aug_path,
                        'gt_path': self.gt_map[gt_key]
                    })
        
        print(f"Dataset loaded. Total pairs: {len(self.valid_pairs)}")

    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, index):
        pair = self.valid_pairs[index]
        
        aug_img = Image.open(pair['aug_path']).convert('RGB')
        gt_img = Image.open(pair['gt_path']).convert('RGB')
        
        aug_tensor = self.transform(aug_img)
        gt_tensor = self.transform(gt_img)
        
        # return aug_tensor, gt_tensor, pair['aug_path'], pair['gt_path']
        return aug_tensor, gt_tensor
