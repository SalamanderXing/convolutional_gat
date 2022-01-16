import torch
import torch.nn as nn
from .GATLayerSpatial import GATLayerSpatial
from .GATLayerTemporal import GATLayerTemporal
from .GATLayerMultiStream import GATLayerMultiStream


class GATMultiHead3D(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        alpha,
        nheads,
        type_,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        mapping_type="linear"
    ):
        super().__init__()
        if type_ == "spatial":
            self.attentions = [
                GATLayerSpatial(
                    in_features=nfeat,
                    out_features=nhid,
                    alpha=alpha,
                    mapping_type=mapping_type,
                    image_width=image_width,
                    image_height=image_height,
                    n_vertices=n_vertices,
                )
                for _ in range(nheads)
            ]
        elif type_ == "temporal":
            self.attentions = [
                GATLayerTemporal(
                    in_features=nfeat,
                    out_features=nhid,
                    alpha=alpha,
                    mapping_type=mapping_type,
                    image_width=image_width,
                    image_height=image_height,
                    n_vertices=n_vertices,
                )
                for _ in range(nheads)
            ]
        elif type_ == "multi_stream":
            self.attentions = [
                GATLayerMultiStream(
                    in_features=nfeat,
                    out_features=nhid,
                    alpha=alpha,
                    mapping_type=mapping_type,
                    image_width=image_width,
                    image_height=image_height,
                    n_vertices=n_vertices,
                )
                for _ in range(nheads)
            ]
        else:
            raise Exception("Type needs to be one of (spatial|temporal)")

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x):
        # N, H, W, T, V = x.size()
        x = torch.cat(tuple(att(x) for att in self.attentions), dim=2)
        return x
