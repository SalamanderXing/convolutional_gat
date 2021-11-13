import torch
import torch.nn as nn
from .GATLayerSpatial import GATLayerSpatial
from .GATLayerTemporal import GATLayerTemporal


class GATMultiHead3D(nn.Module):
    def __init__(self, nfeat, nhid, alpha, nheads, type_):
        super().__init__()
        if type_ == "spatial":
            self.attentions = [
                GATLayerSpatial(in_features=nfeat, out_features=nhid, alpha=alpha)
                for _ in range(nheads)
            ]
        elif type_ == "temporal":
            self.attentions = [
                GATLayerTemporal(in_features=nfeat, out_features=nhid, alpha=alpha)
                for _ in range(nheads)
            ]
        else:
            raise Exception("Type needs to be one of (spatial|temporal)")

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x):
        # N, H, W, T, V = x.size()
        x = torch.cat([att(x) for att in self.attentions], dim=2)
        return x
