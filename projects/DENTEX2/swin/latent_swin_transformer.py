import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.models.backbones import SwinTransformer


@MODELS.register_module()
class LatentSwinTransformer(SwinTransformer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x, latents = x

        x = super().forward(x)
        x = self.gap(x[-1]).view(x[-1].size(0), -1)

        return (latents, x)
