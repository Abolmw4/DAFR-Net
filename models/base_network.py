import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Any

class BaseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
    
    def forwrad(self, **kwargs) -> Any:
        pass
    
    def flops(self) -> float:
        pass
