import torch as t
import torch.nn as nn
from .GAT3D.GATMultiHead3D import GATMultiHead3D

# from .GAT3D.GATMultistream import GATMultiHead3D


class SpatialModel(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):

        super().__init__()
        self.mapping_type = mapping_type
        self.hidden_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=3,
            type="spatial",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )
        self.output_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=1,
            type_="spatial",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        # x = self.output_layer(x)
        return x


class TemporalModel(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.hidden_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=3,
            type_="temporal",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )
        self.output_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=1,
            type_="temporal",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        # x = self.output_layer(x)
        return x


class TemporalModel4h(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.hidden_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=4,
            type_="temporal",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        return x


class TemporalModel2l(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.hidden_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=3,
            type_="temporal",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )
        self.output_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=3,
            type_="temporal",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
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


# class TemporalModel(nn.Module):
#     def __init__(
#         self,
#         *,
#         image_width: int,
#         image_height: int,
#         n_vertices: int,
#         time_steps: int = 4,
#         mapping_type="linear",
#     ):
#         super().__init__()
#         self.mapping_type = mapping_type
#         self.hidden_layer = GATMultiHead3D(
#             nfeat=time_steps,
#             nhid=time_steps,
#             alpha=0.2,
#             nheads=1,
#             type_="temporal",
#             mapping_type=mapping_type,
#             image_height=image_height,
#             image_width=image_width,
#             n_vertices=n_vertices,
#         )
#         self.output_layer = GATMultiHead3D(
#             nfeat=time_steps,
#             nhid=time_steps,
#             alpha=0.2,
#             nheads=1,
#             type_="temporal",
#             mapping_type=mapping_type,
#             image_height=image_height,
#             image_width=image_width,
#             n_vertices=n_vertices,
#         )

#     def forward(self, x):
#         x = self.hidden_layer(x)
#         x = self.output_layer(x)
#         return x


class MultiStreamModel(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.hidden_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=1,
            type_="multi_stream",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )
        self.output_layer = GATMultiHead3D(
            nfeat=time_steps,
            nhid=time_steps,
            alpha=0.2,
            nheads=1,
            type_="multi_stream",
            mapping_type=mapping_type,
            image_height=image_height,
            image_width=image_width,
            n_vertices=n_vertices,
        )

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x
