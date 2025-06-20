import torch
from torch import nn

def attention_map(features: torch.Tensor) -> torch.Tensor:
    """Compute simple attention maps by channel-wise sum of absolute activations."""
    return features.abs().sum(dim=1, keepdim=True)

def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix of feature maps."""
    b, c, h, w = features.size()
    f = features.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)

class AttentionHook:
    """Utility to capture intermediate attention maps."""
    def __init__(self):
        self.maps = []

    def __call__(self, module, input, output):
        self.maps.append(attention_map(output).detach())

    def clear(self):
        self.maps.clear()
