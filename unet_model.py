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
        attention_type: str,
        mapping_type: str = "conv",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.unet = SmaAt_UNet(n_channels=4, n_classes=4)

    def forward(self, x):
        acc = []
        x = x.permute(4, 0, 3, 1, 2)
        for i in range(x.shape[0]):  # should be vertex dim
            acc.append(self.unet(x[i, :, :, :, :]))
        result = t.stack(acc)
        result = result.permute(1, 3, 4, 2, 0)
        return result
