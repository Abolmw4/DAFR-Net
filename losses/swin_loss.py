import torch.nn as nn
import torch
from torchvision.models import vgg16

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        pass
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor) ->torch.Tensor:
        pass