import torch as t
import torch.nn as nn
from .GAT3D.GATMultiHead3D import GATMultiHead3D


class SpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = GATMultiHead3D(
            nfeat=4, nhid=2, alpha=0.2, nheads=3, type_="spatial"
        )
        self.output_layer = GATMultiHead3D(
            nfeat=6, nhid=4, alpha=0.2, nheads=1, type_="spatial"
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = GATMultiHead3D(
            nfeat=4, nhid=4, alpha=0.2, nheads=1, type_="temporal"
        )
        self.output_layer = GATMultiHead3D(
            nfeat=4, nhid=4, alpha=0.2, nheads=1, type_="temporal"
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


class ConvGAT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
