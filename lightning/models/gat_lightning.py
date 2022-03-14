import torch as t
from ...GAT3D.GATMultistream import Model
from .regression_lightning import Precip_regression_base


class GATLightning(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        # self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)
        self.model = Model(
            image_width=80,
            image_height=80,
            n_vertices=6,
            attention_type="temporal",
            mapping_type="conv",
        )

    def forward(self, x):
        return self.model(x)
