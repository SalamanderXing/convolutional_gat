import torch as t
from torch import nn
import torch.nn.functional as F
from .GAT3D.smaat_unet.SmaAt_UNet import SmaAt_UNet
import ipdb


class UnetModel(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        mapping_type: str = "conv"
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.unet = SmaAt_UNet(n_channels=4, n_classes=4)

    def forward(self, x):
        return self.unet(x)
