import torch
import os
import requests
from typing import Callable

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def verify_model(model):
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            print(f"TRAIN  | {param.numel():>10,} | {name}")
        else:
            frozen_count += param.numel()
            print(f"FREEZE | {param.numel():>10,} | {name}")

    total = trainable_count + frozen_count

    print("\n====================")
    print(f"Trainable params: {trainable_count:,}")
    print(f"Frozen params:    {frozen_count:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable_count / total:.2f}%")
    
    
def load_model(model: Callable, weight_src: str="/workspace/model_zoo/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"):
        pretrained_model = torch.load(weight_src)
        model.load_state_dict(pretrained_model["params"] if "params" in pretrained_model.keys() else pretrained_model, strict=False)
        return model
