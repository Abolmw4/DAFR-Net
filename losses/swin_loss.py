import torch.nn as nn
from typing import Tuple, List
from torchvision.models import vgg19, VGG19_Weights
import torch
from pytorch_msssim import SSIM

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers_id: Tuple[int]=(3, 8, 17, 26, 35)) -> None:
        super().__init__()
        vgg = vgg19(VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList() 
        prev = 0
        
        for idx in layers_id:
            self.layers.append(vgg[prev:idx+1])
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        
        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for block in self.layers:
            x = block(x)
            features.append(x)
        return features
    
class VGGNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
        
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids: Tuple[int]=(3, 8, 17, 26, 35), weights: Tuple[float]=(1.0, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layer_ids)
        self.norm = VGGNormalize()
        
        self.weights = weights
        self.criterion = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.norm(pred)
        target = self.norm(target)
        
        pred_feature = self.vgg(pred)
        target_feature = self.vgg(target)
        
        loss = 0
        for w, pf, tf in zip(self.weights, pred_feature, target_feature):
            loss += w * self.criterion(pf, tf.detach())
        return loss

class PixelLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(pred, target)
        return loss
    

class SSIMLoss(nn.Module):
    def __init__(self, data_range: float = 1.0):
        super().__init__()

        self.ssim = SSIM(data_range=data_range, size_average=True, channel=3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return 1.0 - self.ssim(pred, target)
