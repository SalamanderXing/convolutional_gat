import torch as t
from ...GAT3D.GATMultistream import GATMultiHead3D
from .regression_lightning import Precip_regression_base

class GATLightning(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)

    def forward(self, x):
        return self.model(x)

